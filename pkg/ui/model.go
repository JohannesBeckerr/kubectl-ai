// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ui

import (
	"io"
	"sync"
)

type Document struct {
	mutex         sync.Mutex
	subscriptions []*subscription
	nextID        uint64

	Blocks []Block
}

func (d *Document) NumBlocks() int {
	return len(d.Blocks)
}

func (d *Document) IndexOf(find Block) int {
	for i, b := range d.Blocks {
		if b == find {
			return i
		}
	}
	return -1
}

func NewDocument() *Document {
	return &Document{
		nextID: 1,
	}
}

type Block interface {
	attached(doc *Document)

	Document() *Document
}

type Subscriber interface {
	DocumentChanged(doc *Document, block Block)
}

type subscription struct {
	doc        *Document
	id         uint64
	subscriber Subscriber
}

func (s *subscription) Close() error {
	s.doc.mutex.Lock()
	defer s.doc.mutex.Unlock()
	s.subscriber = nil
	return nil
}

func (d *Document) AddSubscription(subscriber Subscriber) io.Closer {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	id := d.nextID
	d.nextID++

	s := &subscription{
		doc:        d,
		id:         id,
		subscriber: subscriber,
	}
	for i, s := range d.subscriptions {
		if s == nil {
			d.subscriptions[i] = s
			return s
		}
	}
	d.subscriptions = append(d.subscriptions, s)
	return s
}

func (d *Document) sendDocumentChangedHoldingLock(b Block) {
	for i, s := range d.subscriptions {
		if s == nil {
			continue
		}
		if s.subscriber == nil {
			d.subscriptions[i] = nil
			continue
		}

		s.subscriber.DocumentChanged(d, b)
	}
}

func (d *Document) AddBlock(block Block) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.Blocks = append(d.Blocks, block)
	block.attached(d)

	d.sendDocumentChangedHoldingLock(block)
}

func (d *Document) blockChanged(block Block) {
	if d == nil {
		return
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.sendDocumentChangedHoldingLock(block)
}
